"use client"
import * as React from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2, Phone, Play, FileText, Check, X, HelpCircle } from "lucide-react"
import { supabase } from "@/lib/supabase"

export default function CampaignPage() {
  const { id } = useParams()
  const [leads, setLeads] = React.useState<any[]>([])
  const [campaign, setCampaign] = React.useState<any>(null)
  const [loading, setLoading] = React.useState(true)
  const [starting, setStarting] = React.useState(false)

  React.useEffect(() => {
    if (id) {
      fetchCampaignData()
      // Real-time subscription
      const channel = supabase
        .channel('table-db-changes')
        .on(
          'postgres_changes',
          {
            event: 'UPDATE',
            schema: 'public',
            table: 'leads',
            filter: `campaign_id=eq.${id}`,
          },
          (payload) => {
            console.log('Update received!', payload)
            setLeads((current) => 
              current.map(lead => lead.id === payload.new.id ? payload.new : lead)
            )
          }
        )
        .subscribe()

      return () => {
        supabase.removeChannel(channel)
      }
    }
  }, [id])

  async function fetchCampaignData() {
    try {
      const { data: camp } = await supabase.from('campaigns').select('*').eq('id', id).single()
      setCampaign(camp)

      const { data: leadsData } = await supabase
        .from('leads')
        .select('*')
        .eq('campaign_id', id)
        .order('id', { ascending: true })
      
      setLeads(leadsData || [])
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const handleStartCampaign = async () => {
    if (!confirm("Are you sure you want to start calling all pending customers?")) return
    setStarting(true)
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765';
      await fetch(`${API_URL}/start-campaign/${id}`, { method: 'POST' })
      alert("Campaign started!")
    } catch (e) {
      alert("Failed to start campaign")
    } finally {
      setStarting(false)
    }
  }

  if (loading) return <div className="flex justify-center p-10"><Loader2 className="animate-spin" /></div>

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">{campaign?.name}</h1>
          <p className="text-muted-foreground">ID: {id}</p>
        </div>
        <Button onClick={handleStartCampaign} disabled={starting || campaign?.status === 'completed'}>
          {starting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
          Start Campaign
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Live Report</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative w-full overflow-auto">
            <table className="w-full caption-bottom text-sm">
              <thead className="[&_tr]:border-b">
                <tr className="border-b">
                  <th className="h-12 px-4 text-left">Customer</th>
                  <th className="h-12 px-4 text-left">Phone</th>
                  <th className="h-12 px-4 text-left">Items</th>
                  <th className="h-12 px-4 text-left">Time</th>
                  <th className="h-12 px-4 text-center">Status</th>
                  <th className="h-12 px-4 text-left">Transcript</th>
                </tr>
              </thead>
              <tbody>
                {leads.map((lead) => (
                  <tr key={lead.id} className="border-b">
                    <td className="p-4">{lead.customer_name}</td>
                    <td className="p-4">{lead.phone_number}</td>
                    <td className="p-4">{lead.order_items}</td>
                    <td className="p-4">{lead.delivery_time}</td>
                    <td className="p-4 text-center">
                      <StatusBadge status={lead.call_status} />
                    </td>
                    <td className="p-4">
                      {lead.transcript ? (
                        <div className="max-w-xs truncate text-xs text-muted-foreground" title={lead.transcript}>
                          {lead.transcript}
                        </div>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    PENDING: "bg-gray-100 text-gray-800",
    CALLED: "bg-blue-100 text-blue-800",
    CONFIRMED: "bg-green-100 text-green-800",
    CANCELLED: "bg-red-100 text-red-800",
    NO_ANSWER: "bg-yellow-100 text-yellow-800",
  }
  
  const icons: Record<string, any> = {
    PENDING: HelpCircle,
    CALLED: Phone,
    CONFIRMED: Check,
    CANCELLED: X,
    NO_ANSWER: HelpCircle,
  }

  const Icon = icons[status] || HelpCircle

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${styles[status] || styles.PENDING}`}>
      <Icon className="w-3 h-3 mr-1" />
      {status}
    </span>
  )
}
